class Machinate < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinate"
  url "https://github.com/jmcguigan10/Machinate/releases/download/v0.1.0/machinate-0.1.0.tar.gz"
  sha256 "2edbef22fd2f22db816d5f28d2a6f16e7c32fea208a9591e2931ca07cdd93e38"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinate", shell_output("#{bin}/macht --help")
  end
end
